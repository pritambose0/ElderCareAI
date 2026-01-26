// Router for Signup Pages
import { useSearchParams } from 'react-router-dom';
import ElderSignupForm from './ElderSignupForm';
import FamilySignupPage from './FamilySignupPage';

const SignupPage = () => {
    const [searchParams] = useSearchParams();
    const role = searchParams.get('role');

    return role === 'family' ? <FamilySignupPage /> : <ElderSignupForm />;
};

export default SignupPage;
